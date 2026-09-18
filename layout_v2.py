"""AutoLayout v2 driver: biologically-structured metabolic map generation.

    python layout_v2.py --model e_coli_core --preview
    python layout_v2.py --model e_coli_core --subsystem "Citric Acid Cycle"
    python layout_v2.py --all

Writes one Escher map per subsystem plus a whole-model map, and prints the
acceptance metrics from layout_algorithm.md S9 for each.

See layout_algorithm.md for the algorithm and CLAUDE.md for how this relates
to the v1 pipeline in process_subsystems.py, which it does not replace yet.
"""

import argparse
import glob
import os
import sys
import traceback

import cobra

from src.layout import metrics, preview, render
from src.layout.compound import compute_cofactor_scores
from src.layout.decompose import clusters
from src.layout.engine import layout_reactions

MODEL_DIR = "data/bigg/models"
DEFAULT_OUT = "layout_output"
AUTHOR = "Tianyu Wu (GitHub: forxhunter)"


def load_model(path):
    if path.endswith(".json"):
        return cobra.io.load_json_model(path)
    return cobra.io.read_sbml_model(path)


def safe_name(name):
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in name)


def subsystems_of(model):
    groups = {}
    for reaction in model.reactions:
        key = (getattr(reaction, "subsystem", "") or "Uncategorized").strip() or "Uncategorized"
        groups.setdefault(key, []).append(reaction)
    return groups


def metabolite_groups(reactions):
    """Cluster key per metabolite: the subsystem most of its reactions sit in.

    Passed to the layered pass so a whole-model map keeps each pathway in one
    place rather than interleaving them during crossing minimisation.
    """
    tally = {}
    for reaction in reactions:
        key = (getattr(reaction, "subsystem", "") or "Uncategorized").strip() or "Uncategorized"
        for metabolite in reaction.metabolites:
            counts = tally.setdefault(metabolite.id, {})
            counts[key] = counts.get(key, 0) + 1
    return {met: max(counts, key=counts.get) for met, counts in tally.items()}


def emit(model, reactions, name, out_dir, want_preview, use_fba, verbose, pitch,
         groups=None):
    result = layout_reactions(model, reactions, name, author=AUTHOR,
                              use_fba=use_fba, verbose=verbose, groups=groups)
    if result is None:
        print(f"  {name}: no drawable structure, skipped")
        return None

    stem = os.path.join(out_dir, safe_name(name))
    render.save(result.escher_map, stem + ".json")
    if want_preview:
        preview.render(result.escher_map, stem + ".png")

    values = metrics.score(result.escher_map, pitch=pitch)
    print(f"  {name}: {values['reactions']} reactions, {values['nodes']} nodes")
    print(metrics.format_report(values))
    return values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", help="model id under data/bigg/models, or a path")
    parser.add_argument("--all", action="store_true", help="process every model in MODEL_DIR")
    parser.add_argument("--subsystem", help="lay out only this subsystem")
    parser.add_argument("--combined", action="store_true",
                        help="also lay out the whole model as one map")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--preview", action="store_true", help="also write a PNG per map")
    parser.add_argument("--no-fba", action="store_true",
                        help="skip pFBA; orient reversible reactions topologically only")
    parser.add_argument("--no-groups", action="store_true",
                        help="do not keep subsystems contiguous in the combined map")
    parser.add_argument("--raw-subsystems", action="store_true",
                        help="use declared subsystems verbatim, skipping the "
                             "constraints.md size limits and community fallback")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.all:
        paths = sorted(glob.glob(os.path.join(MODEL_DIR, "*.json"))
                       + glob.glob(os.path.join(MODEL_DIR, "*.xml")))
        seen, unique = set(), []
        for path in paths:
            stem = os.path.splitext(os.path.basename(path))[0]
            if stem not in seen:
                seen.add(stem)
                unique.append(path)
        paths = unique
    elif args.model:
        if os.path.exists(args.model):
            paths = [args.model]
        else:
            candidates = [os.path.join(MODEL_DIR, args.model + ext) for ext in (".json", ".xml")]
            paths = [p for p in candidates if os.path.exists(p)][:1]
            if not paths:
                parser.error(f"model not found: {args.model}")
    else:
        parser.error("pass --model or --all")

    from src.layout.engine import LAYER_GAP

    for path in paths:
        model_id = os.path.splitext(os.path.basename(path))[0]
        out_dir = os.path.join(args.out, model_id)
        os.makedirs(out_dir, exist_ok=True)
        print(f"Model {model_id}")

        try:
            model = load_model(path)
        except Exception as exc:
            print(f"  failed to load: {exc}")
            continue

        if args.raw_subsystems:
            groups = subsystems_of(model)
        else:
            groups = clusters(model, compute_cofactor_scores(model))
        targets = ({args.subsystem: groups[args.subsystem]}
                   if args.subsystem and args.subsystem in groups else groups)
        if args.subsystem and args.subsystem not in groups:
            print(f"  no such subsystem; available: {sorted(groups)}")
            continue
        print(f"  {len(groups)} clusters")

        for name, reactions in sorted(targets.items()):
            try:
                emit(model, reactions, name, out_dir, args.preview,
                     not args.no_fba, not args.quiet, LAYER_GAP)
            except Exception as exc:
                print(f"  {name}: FAILED {exc}")
                traceback.print_exc()

        if args.combined and not args.subsystem:
            try:
                emit(model, list(model.reactions), f"{model_id}_Combined", out_dir,
                     args.preview, not args.no_fba, not args.quiet, LAYER_GAP,
                     groups=None if args.no_groups else metabolite_groups(model.reactions))
            except Exception as exc:
                print(f"  combined: FAILED {exc}")
                traceback.print_exc()

    return 0


if __name__ == "__main__":
    sys.exit(main())
