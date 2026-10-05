"""AutoLayout v2 driver: biologically-structured metabolic map generation.

    python layout_v2.py --model e_coli_core --preview
    python layout_v2.py --model e_coli_core --subsystem "Citric Acid Cycle"
    python layout_v2.py --all

Writes one Escher map per subsystem plus a whole-model map, and prints the
acceptance metrics from layout_algorithm.md S9 for each.

See layout_algorithm.md for the algorithm and how this relates
to the v1 pipeline in process_subsystems.py, which it does not replace yet.
"""

import argparse
import glob
import hashlib
import os
import re
import shutil
import sys
import traceback

import cobra

from src.layout import metrics, pdfout, preview, render, svgout
from src.layout.compose import build_meta_graph, compose
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


# Longest file stem. A map's title is now a sentence -- "Transport and
# exchange: Transport, Inner Membrane (amino acids and peptides)" -- and used
# whole as a file name it pushed paths past Windows' 260-character limit. The
# title lives in the map; the file name only has to be unique and readable.
MAX_STEM = 60


def file_stem(name):
    """A short, unique, filesystem-safe stem for a map titled `name`."""
    stem = re.sub(r"_+", "_", safe_name(name)).strip("_")
    if len(stem) <= MAX_STEM:
        return stem
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:6]
    return stem[:MAX_STEM - 7].rstrip("_") + "_" + digest


def merge_by_function(groups, max_size=None):
    """Consolidate clusters that do the same kind of chemistry into one map.

    Packing 300 pathway tiles onto a canvas leaves 300 separate little drawings
    however neatly they are arranged. Merging by biological function instead
    gives one drawing per metabolic superclass -- all of lipid metabolism as a
    single connected map, all of amino acid metabolism as another -- which is
    how `templates/t1` and `t2` are organised and what makes them navigable.

    `taxonomy.classify` files each cluster under a KEGG BRITE top-level
    category, by its name when that says and by what its reactions are when it
    does not.

    A superclass that blows past `max_size` is split into pages, each titled
    for the pathways on it, so the reader still sees one region and can tell
    its pages apart.
    """
    from src.layout import naming, taxonomy
    from src.layout.decompose import load_kegg_mapping

    mapping = load_kegg_mapping()
    pooled = {}
    for name, reactions in groups.items():
        label = taxonomy.classify(name, reactions, mapping)
        pooled.setdefault(label, []).append((name, reactions))

    out = {}
    for label, members in pooled.items():
        total = sum(len(r) for _, r in members)
        if not max_size or total <= max_size:
            out[label] = [r for _, rs in members for r in rs]
            continue

        # Bin-pack whole pathways, never slicing one.
        #
        # Cutting the merged reaction list every `max_size` entries is the
        # obvious way to do this and it is wrong: the cut lands in the middle of
        # whichever pathway happens to straddle it, so glycolysis ends up half
        # in sheet 2 and half in sheet 3 for no reason a reader can see. Keeping
        # the pathway as the atom costs a little packing efficiency and keeps
        # every pathway whole.
        #
        # Pathways are placed largest first, and a piece of a split pathway goes
        # on a page that already holds its siblings when one has room. A page is
        # then one large pathway plus the small ones that fill its gaps, which
        # is something a title can say.
        family = {name: naming.split_piece(name)[0] for name, _ in members}
        family_size = {}
        for name, reactions in members:
            family_size[family[name]] = family_size.get(family[name], 0) + len(reactions)
        order = sorted(members, key=lambda kv: (-family_size[family[kv[0]]],
                                                family[kv[0]], -len(kv[1]), kv[0]))
        bins = []
        for name, reactions in order:
            fits = [b for b in bins if b["size"] + len(reactions) <= max_size]
            same = [b for b in fits if family[name] in b["families"]]
            target = (same or fits or [None])[0]
            if target is None:
                target = {"members": [], "size": 0, "families": set()}
                bins.append(target)
            target["members"].append((name, reactions))
            target["size"] += len(reactions)
            target["families"].add(family[name])

        if len(bins) == 1:
            out[label] = [r for _, rs in bins[0]["members"] for r in rs]
            continue
        titles = naming.title_pages(
            label, [[(name, len(rs)) for name, rs in b["members"]] for b in bins])
        for title, b in zip(titles, bins):
            out[title] = [r for _, rs in b["members"] for r in rs]
    return out


def species_canvas(model, model_id, targets, tiles):
    """The whole model on one canvas, regions by KEGG superclass."""
    from src.layout import organisms, taxonomy
    from src.layout.canvas import compose_canvas
    from src.layout.decompose import load_kegg_mapping

    mapping = load_kegg_mapping()
    labels = {name: taxonomy.classify(name, targets[name], mapping) for name, _ in tiles}
    meta = build_meta_graph({n: r for n, r in targets.items()},
                            compute_cofactor_scores(model))
    _, species, common = organisms.describe(model_id)
    title = model_id + (f" - {species}" if species else "") + (f" ({common})" if common else "")
    return compose_canvas(tiles, labels, meta, title, author=AUTHOR)


def subsystems_of(model):
    from src.layout.decompose import declared_subsystems
    return declared_subsystems(model.reactions)


def metabolite_groups(reactions):
    """Cluster key per metabolite: the subsystem most of its reactions sit in.

    Passed to the layered pass so a whole-model map keeps each pathway in one
    place rather than interleaving them during crossing minimisation.
    """
    from src.layout.decompose import clean_subsystem

    tally = {}
    for reaction in reactions:
        key = clean_subsystem(getattr(reaction, "subsystem", "")) or "Uncategorized"
        for metabolite in reaction.metabolites:
            counts = tally.setdefault(metabolite.id, {})
            counts[key] = counts.get(key, 0) + 1
    return {met: max(counts, key=counts.get) for met, counts in tally.items()}


def emit(model, reactions, name, out_dir, want_preview, use_fba, verbose, pitch,
         groups=None, write=True):
    """Lay out one cluster; `write=False` keeps it in memory as a tile only.

    A whole-model map still needs every cluster drawn, but it does not need
    every cluster *written*: Recon3D emits 300-odd per-pathway files that nobody
    opens, and they then go stale the moment cluster naming changes.
    """
    result = layout_reactions(model, reactions, name, author=AUTHOR,
                              use_fba=use_fba, verbose=verbose, groups=groups)
    if result is None:
        print(f"  {name}: no drawable structure, skipped")
        return None
    if write:
        save_map(result.escher_map, out_dir, name, want_preview, pitch, verbose)
    return result.escher_map


def save_map(escher_map, out_dir, name, want_preview, pitch, verbose=True):
    stem = os.path.join(out_dir, file_stem(name))
    render.save(escher_map, stem + ".json")
    if want_preview:
        preview.render(escher_map, stem + ".png")
        # SVG alongside the PNG: a raster has one resolution, and these maps are
        # read by zooming into a corner rather than at fit-to-page. The vector
        # copy stays sharp at any magnification, prints at any DPI, and costs
        # about as much to write as the PNG does.
        svgout.render(escher_map, stem + ".svg")
        # PDF as well, because it is the one of the three a reader can drop
        # into a manuscript: vector like the SVG, but LaTeX and Word read it
        # directly, and it is the smallest of the three on disk.
        pdfout.render(escher_map, stem + ".pdf", width_mm=180.0)
    values = metrics.score(escher_map, pitch=pitch)
    if verbose:
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
                             "layout_algorithm.md size limits and community fallback")
    parser.add_argument("--no-clean", dest="clean", action="store_false",
                        default=True,
                        help="keep maps from previous runs. Cleaning is the "
                             "default: cluster names change between runs, so a "
                             "rename leaves the old file behind and the "
                             "directory silently becomes a union of several "
                             "generations of contradictory maps.")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--canvas", action="store_true",
                        help="also draw the whole model on one densely packed "
                             "canvas (<model>_Canvas), planned on the pathways' "
                             "real shapes rather than their bounding boxes")
    parser.add_argument("--canvas-only", action="store_true",
                        help="write only the one-canvas map. Implies --canvas.")
    parser.add_argument("--group-function", action="store_true",
                        help="merge clusters of the same metabolic superclass "
                             "into one map each (lipid, amino acid, ...), "
                             "instead of emitting one tile per pathway")
    parser.add_argument("--combined-only", action="store_true",
                        help="write only the whole-model map, not one file per "
                             "cluster. Implies --combined.")
    parser.add_argument("--max-cluster", type=int, default=None,
                        help="largest cluster in reactions (default 60). Raising "
                             "it trades per-tile readability for a whole-model "
                             "map made of a few large regions instead of "
                             "hundreds of postage stamps.")
    parser.add_argument("--min-cluster", type=int, default=None,
                        help="smallest cluster in reactions (default 6)")
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
        if args.clean and os.path.isdir(out_dir):
            shutil.rmtree(out_dir, ignore_errors=True)
        os.makedirs(out_dir, exist_ok=True)
        before = {f for f in os.listdir(out_dir) if f.endswith(".json")}
        written = set()
        print(f"Model {model_id}")

        try:
            model = load_model(path)
        except Exception as exc:
            print(f"  failed to load: {exc}")
            continue

        if args.raw_subsystems:
            groups = subsystems_of(model)
        else:
            from src.layout import decompose
            size_limits = {}
            if args.max_cluster:
                size_limits["max_size"] = args.max_cluster
            if args.min_cluster:
                size_limits["min_size"] = args.min_cluster
            groups = clusters(model, compute_cofactor_scores(model), **size_limits)
            if args.group_function:
                pathway_count = len(groups)
                groups = merge_by_function(groups, max_size=args.max_cluster)
                print(f"  merged {pathway_count} pathway clusters into "
                      f"{len(groups)} functional maps")
        targets = ({args.subsystem: groups[args.subsystem]}
                   if args.subsystem and args.subsystem in groups else groups)
        if args.subsystem and args.subsystem not in groups:
            print(f"  no such subsystem; available: {sorted(groups)}")
            continue
        print(f"  {len(groups)} clusters")

        tiles = []
        for name, reactions in sorted(targets.items()):
            try:
                tile = emit(model, reactions, name, out_dir, args.preview,
                            not args.no_fba, not args.quiet, LAYER_GAP,
                            write=not (args.combined_only or args.canvas_only))
                written.add(file_stem(name) + ".json")
                if tile is not None:
                    tiles.append((name, tile))
            except Exception as exc:
                print(f"  {name}: FAILED {exc}")
                traceback.print_exc()

        if (args.combined or args.combined_only) and not args.subsystem and tiles:
            # Meta-tiling: reuse the per-cluster drawings and lay the tiles out
            # with the same layered pass. Drawing the whole model in one go
            # instead is what produces an unreadable 2.6-crossings-per-edge map.
            try:
                meta = build_meta_graph({n: r for n, r in targets.items()},
                                        compute_cofactor_scores(model))
                combined = compose(tiles, meta, f"{model_id}_Combined", author=AUTHOR)
                if combined is not None:
                    save_map(combined, out_dir, f"{model_id}_Combined",
                             args.preview, LAYER_GAP, not args.quiet)
                    written.add(file_stem(f"{model_id}_Combined") + ".json")
            except Exception as exc:
                print(f"  combined: FAILED {exc}")
                traceback.print_exc()

        if (args.canvas or args.canvas_only) and not args.subsystem and tiles:
            try:
                sheet = species_canvas(model, model_id, targets, tiles)
                if sheet is not None:
                    save_map(sheet, out_dir, f"{model_id}_Canvas",
                             args.preview, LAYER_GAP, not args.quiet)
                    written.add(file_stem(f"{model_id}_Canvas") + ".json")
                    if not args.quiet:
                        blank = metrics.blank_space(sheet)
                        print(f"    blank share {blank['blank_share']:.3f}, largest "
                              f"blank {blank['largest_blank_share']:.3f} of the canvas")
            except Exception as exc:
                print(f"  canvas: FAILED {exc}")
                traceback.print_exc()

        # Cluster names change between runs, so a rename leaves the old file
        # behind and the directory becomes the union of every run that ever
        # wrote to it. Anything that then reads the directory -- a metric sweep,
        # the map index, a reviewer -- scores a mixture of code versions. This
        # has silently corrupted measurements more than once, so say so loudly
        # rather than deleting a user's files without being asked.
        stale = before - written
        if stale:
            print(f"  WARNING: {len(stale)} stale map(s) from earlier runs remain in "
                  f"{out_dir}")
            print(f"           this run wrote {len(written)}; anything reading that "
                  f"directory will score a mixture")
            print("           re-run with --clean to make the directory this run only")

    return 0


if __name__ == "__main__":
    sys.exit(main())
