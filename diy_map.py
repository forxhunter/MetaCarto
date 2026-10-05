"""Draw your own map: choose what goes in, get one Escher map and an SVG.

The published maps cut a model where the pipeline thinks pathways end. A
reader often wants a different cut -- glycolysis with the pentose phosphate
pathway beside it, everything within two steps of glutamate, the reactions a
flux experiment touched. This builds that map with the same engine and holds
it to the same rules as the published ones: no text on a node, an edge or
other text; no reaction drawn on top of another; no large empty areas.

Selection (any combination; the map is the union):

  --pathway  "Citric Acid Cycle"      pathway or subsystem name, substring or glob
  --superclass "Carbohydrate*"        KEGG superclass the pathway is filed under
  --reaction PDH,CS,AKGDH             reaction ids (globs allowed)
  --metabolite akg --radius 2         everything within N reactions of a compound
  --search glutamate                  reaction or metabolite names containing it
  --from-file picks.txt               reaction ids, one per line, or a .json list

then optionally

  --connect 2                         add up to N reactions to join separate pieces
  --exclude "*transport*"             drop matching pathways or reactions
  --no-boundary                       drop exchange, demand, sink and biomass steps

A small selection is drawn as one connected drawing, so the links between the
chosen pathways are drawn as edges. A large one is drawn pathway by pathway
and packed onto one canvas, the way the whole-model canvas is.

  python diy_map.py --model e_coli_core --pathway Glycolysis --pathway "Pentose*"
  python diy_map.py --model iML1515 --metabolite glu__L --radius 1 --name "Glutamate hub"
  python diy_map.py --model e_coli_core --list pathways
"""

import argparse
import fnmatch
import json
import os
import sys

import networkx as nx

from layout_v2 import AUTHOR, MODEL_DIR, MODEL_EXTENSIONS, file_stem, load_model, model_stem
from src.layout import identity, metrics, pdfout, preview, render, svgout, taxonomy
from src.layout.canvas import compose_canvas, cross_overlaps, organisation
from src.layout.compose import build_meta_graph
from src.layout.compound import NEVER_PRIMARY, compute_cofactor_scores
from src.layout.decompose import (clean_subsystem, clusters, is_boundary_cluster,
                                  is_boundary_reaction, load_kegg_mapping, merge_small)
from src.layout.engine import LAYER_GAP, layout_reactions

DEFAULT_OUT = os.path.join("data", "diy_maps")
SINGLE_MAX = 80            # up to this many reactions, one connected drawing
CLUSTER_MAX = 120          # per-pathway drawing size on a canvas, as published
CLUSTER_MIN = 6
CURRENCY_CUTOFF = 0.5


# --------------------------------------------------------------------------
# what the model offers
# --------------------------------------------------------------------------

class Catalogue:
    """The model's pathways as the published maps cut them, and their superclasses."""

    def __init__(self, model):
        self.model = model
        self.scores = compute_cofactor_scores(model)
        self.mapping = load_kegg_mapping()
        self.pathways = clusters(model, self.scores, max_size=CLUSTER_MAX)
        self.superclass = {name: taxonomy.classify(name, reactions, self.mapping)
                           for name, reactions in self.pathways.items()}
        self.pathway_of = {r.id: name for name, reactions in self.pathways.items()
                           for r in reactions}

    def currency(self, metabolite):
        return (self.scores.get(metabolite.id, 0.0) >= CURRENCY_CUTOFF
                or identity.canonical(metabolite) in NEVER_PRIMARY)


def _matches(pattern, text):
    """Glob when the pattern has wildcards, case-insensitive substring otherwise."""
    pattern, text = pattern.lower(), str(text or "").lower()
    if any(c in pattern for c in "*?["):
        return fnmatch.fnmatchcase(text, pattern)
    return pattern in text


def _split(values):
    out = []
    for value in values or ():
        out.extend(v.strip() for v in value.split(",") if v.strip())
    return out


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------

def _seed_metabolites(catalogue, query):
    """Metabolites named by `query`: an id, an id without compartment, or a name."""
    model = catalogue.model
    if query in model.metabolites:
        return [model.metabolites.get_by_id(query)]
    by_species = [m for m in model.metabolites if identity.species(m) == query]
    if by_species:
        return by_species
    exact = [m for m in model.metabolites if (m.name or "").lower() == query.lower()]
    return exact or [m for m in model.metabolites if _matches(query, m.name)]


def neighbourhood(catalogue, seeds, radius):
    """Reactions within `radius` steps of `seeds`, walking only through chemistry.

    Walking through ATP or water reaches half the model in one step, so the
    walk passes through a metabolite only when it is not currency.
    """
    frontier, seen_mets, chosen = list(seeds), set(m.id for m in seeds), []
    picked = set()
    for _ in range(max(radius, 1)):
        next_frontier = []
        for metabolite in frontier:
            for reaction in sorted(metabolite.reactions, key=lambda r: r.id):
                if reaction.id in picked:
                    continue
                picked.add(reaction.id)
                chosen.append(reaction)
                for other in reaction.metabolites:
                    if other.id not in seen_mets and not catalogue.currency(other):
                        seen_mets.add(other.id)
                        next_frontier.append(other)
        frontier = next_frontier
    return chosen


def _reaction_graph(catalogue, reactions):
    """Reactions joined when they share a non-currency metabolite."""
    graph = nx.Graph()
    by_met = {}
    for reaction in reactions:
        graph.add_node(reaction.id)
        for metabolite in reaction.metabolites:
            if not catalogue.currency(metabolite):
                by_met.setdefault(metabolite.id, []).append(reaction.id)
    for members in by_met.values():
        for a, b in zip(members, members[1:]):
            graph.add_edge(a, b)
    return graph


def connect(catalogue, selected, limit):
    """Add the fewest reactions (at most `limit` per gap) that join the pieces.

    A hand-picked set is often two pathways that meet through a reaction
    nobody thought to pick. Drawn without it they are two islands and the
    point of putting them on one map -- that they connect -- is lost.
    """
    model = catalogue.model
    whole = _reaction_graph(catalogue, model.reactions)
    chosen = {r.id for r in selected}
    added = []
    for _ in range(len(chosen)):
        sub = whole.subgraph(chosen)
        pieces = sorted(nx.connected_components(sub), key=len, reverse=True)
        if len(pieces) < 2:
            break
        main = pieces[0]
        best = None
        for piece in pieces[1:]:
            lengths, paths = nx.multi_source_dijkstra(whole, set(piece), cutoff=limit + 1)
            targets = [(lengths[n], n) for n in main if n in lengths]
            if targets:
                d, target = min(targets)
                if best is None or d < best[0]:
                    best = (d, paths[target])
        if best is None:
            break
        bridge = [rid for rid in best[1] if rid not in chosen]
        if not bridge:
            break
        chosen.update(bridge)
        added.extend(bridge)
    return [model.reactions.get_by_id(rid) for rid in added]


def select(catalogue, args):
    """Reactions to draw, in a stable order, and why each was picked."""
    model = catalogue.model
    reasons = {}

    def take(reactions, why):
        for reaction in reactions:
            reasons.setdefault(reaction.id, why)

    for pattern in args.pathway or ():
        names = [n for n in catalogue.pathways if _matches(pattern, n)]
        take([r for n in names for r in catalogue.pathways[n]], f"pathway {pattern}")
        # The model's own subsystem names, which a reader knows from the paper.
        take([r for r in model.reactions
              if _matches(pattern, clean_subsystem(getattr(r, "subsystem", "")))],
             f"subsystem {pattern}")
    for pattern in args.superclass or ():
        names = [n for n, label in catalogue.superclass.items() if _matches(pattern, label)]
        take([r for n in names for r in catalogue.pathways[n]], f"superclass {pattern}")
    for pattern in _split(args.reaction):
        take([r for r in model.reactions if r.id == pattern
              or (any(c in pattern for c in "*?[") and _matches(pattern, r.id))],
             f"reaction {pattern}")
    for path in args.from_file or ():
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        ids = json.loads(text) if path.endswith(".json") else text.split()
        if isinstance(ids, dict):
            ids = ids.get("reactions", [])
        take([model.reactions.get_by_id(i) for i in ids if i in model.reactions],
             f"file {os.path.basename(path)}")
    for query in args.metabolite or ():
        seeds = _seed_metabolites(catalogue, query)
        if not seeds:
            print(f"  no metabolite matches {query!r}")
        take(neighbourhood(catalogue, seeds, args.radius), f"within {args.radius} of {query}")
    for pattern in args.search or ():
        take([r for r in model.reactions
              if _matches(pattern, r.id) or _matches(pattern, r.name)
              or any(_matches(pattern, m.name) for m in r.metabolites
                     if not catalogue.currency(m))],
             f"search {pattern}")

    def excluded(reaction):
        if args.no_boundary and is_boundary_reaction(reaction, catalogue.scores):
            return True
        for pattern in args.exclude or ():
            if (_matches(pattern, reaction.id) or _matches(pattern, reaction.name)
                    or _matches(pattern, catalogue.pathway_of.get(reaction.id, ""))):
                return True
        return False

    chosen = [r for r in model.reactions if r.id in reasons and not excluded(r)]
    if args.connect and chosen:
        bridge = [r for r in connect(catalogue, chosen, args.connect) if not excluded(r)]
        take(bridge, "connects the selection")
        ids = {r.id for r in chosen} | {r.id for r in bridge}
        chosen = [r for r in model.reactions if r.id in ids]
    return chosen, reasons


# --------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------

def plan(catalogue, chosen, single_max):
    """{drawing name: reactions}: one drawing, or one per pathway for a canvas."""
    by_pathway = {}
    for reaction in chosen:
        by_pathway.setdefault(catalogue.pathway_of.get(reaction.id, "Selected reactions"),
                              []).append(reaction)
    if len(chosen) <= single_max or len(by_pathway) == 1:
        return by_pathway, True
    boundary = {name for name, reactions in by_pathway.items()
                if is_boundary_cluster(reactions, catalogue.scores)}
    return merge_small(by_pathway, catalogue.scores, min_size=CLUSTER_MIN,
                       max_size=CLUSTER_MAX, boundary=boundary), False


def draw(catalogue, title, parts, single, use_fba=True, verbose=False):
    model = catalogue.model
    if single:
        reactions = [r for rs in parts.values() for r in rs]
        # Keep each chosen pathway in one place through crossing reduction.
        groups = None
        if len(parts) > 1:
            pathway = {r.id: name for name, rs in parts.items() for r in rs}
            groups = {}
            for reaction in reactions:
                for metabolite in reaction.metabolites:
                    groups.setdefault(metabolite.id, pathway[reaction.id])
        result = layout_reactions(model, reactions, title, author=AUTHOR,
                                  use_fba=use_fba, verbose=verbose, groups=groups)
        if result is None:
            return None, [], {}
        render.annotate_pathways(result.escher_map, sorted(parts.items()))
        return result.escher_map, [], {}

    tiles = []
    for name, reactions in sorted(parts.items()):
        result = layout_reactions(model, reactions, name, author=AUTHOR,
                                  use_fba=use_fba, verbose=verbose)
        if result is not None:
            render.annotate_pathways(result.escher_map, [(name, reactions)])
            tiles.append((name, result.escher_map))
    labels = {name: (catalogue.superclass.get(name)
                     or taxonomy.classify(name, parts[name], catalogue.mapping))
              for name, _ in tiles}
    meta = build_meta_graph(parts, catalogue.scores)
    sheet = compose_canvas(tiles, labels, meta, title, author=AUTHOR)
    return sheet, tiles, (labels, meta)


GATES = (
    # (metric, limit, what it protects)
    ("label_overlaps", 0, "text on text"),
    ("label_on_node", 0, "text on a node"),
    ("label_on_edge", 0, "text on an edge"),
)


def check(escher_map, tiles, extra):
    """The published maps' criteria, measured on this one. Returns (report, ok)."""
    values = metrics.score(escher_map, pitch=LAYER_GAP)
    overlaps = metrics.reaction_overlaps(escher_map)
    blank = metrics.blank_space(escher_map)
    lines = [metrics.format_report(values)]
    for kind in ("node_on_node", "edge_on_node", "edge_on_edge"):
        flag = "!" if kind == "node_on_node" and overlaps[kind + "_unrelated"] else " "
        lines.append(f"  {flag} {kind:22} {overlaps[kind]} "
                     f"({overlaps[kind + '_unrelated']} between unrelated reactions)")
    lines += [f"    blank_share            {blank['blank_share']:.3f}",
              f"    largest_blank_share    {blank['largest_blank_share']:.3f}"]
    # Reactions that share a metabolite may run together for a stretch; two
    # that share nothing must never be drawn on one another.
    ok = all(values[key] <= limit for key, limit, _ in GATES)
    ok = ok and overlaps["node_on_node_unrelated"] == 0
    if tiles:
        labels, meta = extra
        names = [name for name, _ in tiles]
        crossed = cross_overlaps(escher_map)
        order = organisation(escher_map, labels, meta, names)
        lines += [f"    cross_overlaps         {crossed}",
                  f"    region_cohesion        {order['region_cohesion']:.3f}",
                  f"    link_ratio             {order['link_ratio']:.3f}"]
        ok = ok and crossed == 0
    return "\n".join(lines), ok


def resolve_model(spec):
    if os.path.exists(spec):
        return spec
    for ext in MODEL_EXTENSIONS:
        path = os.path.join(MODEL_DIR, spec + ext)
        if os.path.exists(path):
            return path
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True,
                        help="model id under data/bigg/models, or a path (.json .xml .yml .mat)")
    parser.add_argument("--list", choices=("pathways", "superclasses"),
                        help="print what can be selected and exit")
    parser.add_argument("--pathway", action="append")
    parser.add_argument("--superclass", action="append")
    parser.add_argument("--reaction", action="append", help="ids, comma separated")
    parser.add_argument("--metabolite", action="append")
    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument("--search", action="append")
    parser.add_argument("--from-file", action="append")
    parser.add_argument("--connect", type=int, default=0, metavar="N",
                        help="join separate pieces with up to N extra reactions each")
    parser.add_argument("--exclude", action="append")
    parser.add_argument("--no-boundary", action="store_true")
    parser.add_argument("--single-max", type=int, default=SINGLE_MAX,
                        help=f"draw as one connected drawing up to this size (default {SINGLE_MAX})")
    parser.add_argument("--name", help="map title (default: built from the selection)")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--pdf", action="store_true")
    parser.add_argument("--png", action="store_true")
    parser.add_argument("--no-fba", action="store_true")
    parser.add_argument("--strict", action="store_true",
                        help="exit 1 when the map breaks a quality gate")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    path = resolve_model(args.model)
    if path is None:
        parser.error(f"model not found: {args.model}")
    model = load_model(path)
    model_id = model_stem(path)
    catalogue = Catalogue(model)

    if args.list == "pathways":
        for name in sorted(catalogue.pathways, key=lambda n: (taxonomy.order_index(
                catalogue.superclass[n]), n)):
            print(f"{len(catalogue.pathways[name]):5d}  {catalogue.superclass[name]:36s}  {name}")
        return 0
    if args.list == "superclasses":
        totals = {}
        for name, label in catalogue.superclass.items():
            totals[label] = totals.get(label, 0) + len(catalogue.pathways[name])
        for label in sorted(totals, key=taxonomy.order_index):
            print(f"{totals[label]:5d}  {label}")
        return 0

    chosen, reasons = select(catalogue, args)
    if not chosen:
        print("Nothing selected. Try --list pathways, or a broader --search.")
        return 1

    parts, single = plan(catalogue, chosen, args.single_max)
    title = args.name or " + ".join(sorted(parts)[:3]) + (
        f" + {len(parts) - 3} more" if len(parts) > 3 else "")
    print(f"{model_id}: {len(chosen)} reactions from {len(parts)} pathway(s), "
          f"{'one drawing' if single else 'a canvas'}")
    escher_map, tiles, extra = draw(catalogue, title, parts, single,
                                    use_fba=not args.no_fba, verbose=not args.quiet)
    if escher_map is None:
        print("The selection has no drawable structure.")
        return 1

    out_dir = os.path.join(args.out, model_id)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, file_stem(title))
    render.save(escher_map, stem + ".json")
    svgout.render(escher_map, stem + ".svg")
    if args.pdf:
        pdfout.render(escher_map, stem + ".pdf", width_mm=180.0)
    if args.png:
        preview.render(escher_map, stem + ".png")
    # What was asked for, so the same map can be drawn again from the model.
    with open(stem + ".selection.json", "w", encoding="utf-8") as handle:
        json.dump({"model": model_id, "title": title,
                   "arguments": {k: v for k, v in vars(args).items() if v not in (None, False, [])},
                   "reactions": [r.id for r in chosen],
                   "why": {r.id: reasons.get(r.id, "") for r in chosen},
                   "pathways": {name: [r.id for r in rs] for name, rs in parts.items()}},
                  handle, indent=1)

    report, ok = check(escher_map, tiles, extra)
    print(f"  wrote {stem}.json and .svg")
    print(report)
    print("  quality gates: " + ("pass" if ok else "FAIL"))
    return 0 if ok or not args.strict else 1


if __name__ == "__main__":
    sys.exit(main())
