"""Does MetaCarto draw the *right* backbone, not just a tidy one?

Why this exists
---------------
The ablation study could not evidence the method's central claim. Primary-
compound reduction picks one substrate/product pair per reaction to be the
drawn backbone, and the cofactor rule exists so that pyruvate -> acetyl-CoA is
chosen over CoA -> acetyl-CoA, which shares 21 carbons and wins on raw
chemistry. Both choices produce a drawing that is equally orthogonal, equally
separated and equally square, so every metric in `metrics.py` scores them the
same and the ablation returned a null result. The metrics measure whether a map
is tidy. They cannot measure whether it is correct.

This does. KEGG's curators decided, for each reaction they drew, which
substrate and which product to connect -- that is what a `<reaction>` element
in a KGML file records, and the currency metabolites are simply not in it. So
KEGG's drawn connection is an independent, human, published statement of what
a reaction's main pair is, for thousands of reactions.

The comparison
--------------
  BiGG reaction --(annotation kegg.reaction)--> R number
  R number --(KGML)--> the substrate and product KEGG drew
  MetaCarto's chosen edge --(annotation kegg.compound)--> C numbers
  agree?

Coverage is the honest limit: in iJO1366, 752 of 2583 reactions carry a KEGG
reaction id and 1354 of 1805 metabolites carry a KEGG compound id, and only a
subset of those R numbers appear in a KGML file that draws them. The measure
applies to the intersection, and the intersection is reported alongside the
result rather than buried.

    python -m src.bench.correctness --model iJO1366
    python -m src.bench.correctness --model iJO1366 --variants full,no_cofactor_tiering
"""

import argparse
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from src.bench import ablate                          # noqa: E402
from src.layout.compound import build_compound_graph, strip_compartment  # noqa: E402


def kegg_drawn_pairs(directory):
    """{R number: {(substrate C, product C), ...}} as KEGG drew them."""
    pairs = {}
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".xml"):
            continue
        try:
            root = ET.parse(os.path.join(directory, name)).getroot()
        except ET.ParseError:
            continue
        compound = {}
        for entry in root.findall("entry"):
            if entry.get("type") == "compound":
                compound[entry.get("id")] = (entry.get("name") or "").replace("cpd:", "")
        for reaction in root.findall("reaction"):
            names = (reaction.get("name") or "").replace("rn:", "").split()
            subs = [compound.get(s.get("id")) for s in reaction.findall("substrate")]
            prods = [compound.get(p.get("id")) for p in reaction.findall("product")]
            subs = [s for s in subs if s]
            prods = [p for p in prods if p]
            for rid in names:
                bucket = pairs.setdefault(rid, set())
                for s in subs:
                    for p in prods:
                        # Unordered: orientation is a separate claim, tested by
                        # the pFBA ablation, not by this one.
                        bucket.add(frozenset((s, p)))
    return pairs


def _annotation(entity, key):
    value = (entity.annotation or {}).get(key)
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)] if value else []


def chosen_pairs(model, reactions, variant):
    """{reaction id: (substrate met id, product met id)} MetaCarto drew."""
    manager = ablate.VARIANTS.get(variant)
    import contextlib
    ctx = (contextlib.nullcontext() if manager is None or variant == "no_fba"
           else manager())
    with ctx:
        cgraph = build_compound_graph(model, reactions)
    out = {}
    for u, v, data in cgraph.D.edges(data=True):
        for rid in data.get("rxns", ()):
            out.setdefault(rid, (u, v))
    return out


def metdraw_pairs(model, reactions, percentile=0.90):
    """The connections MetDraw would draw, reconstructed from its description.

    MetDraw (Bioinformatics 2014) "identif[ies] major metabolites that are more
    highly connected than other metabolites in the same subsystem", clones
    those, and draws the rest. It does not choose a single backbone pair the
    way primary-compound reduction does -- it draws every remaining
    substrate/product connection -- so the honest comparison is not
    agree/disagree but precision and recall against what KEGG drew.

    The rule is connectivity and nothing else: no conserved-moiety chemistry,
    no curated carrier list. `percentile` is the cloning threshold, taken over
    the degree distribution of the reaction set being drawn, which is what
    "more highly connected than other metabolites in the same subsystem"
    means.
    """
    degree = {}
    for rxn in reactions:
        for met in rxn.metabolites:
            degree[met.id] = degree.get(met.id, 0) + 1
    if not degree:
        return {}
    ordered = sorted(degree.values())
    cutoff = ordered[min(len(ordered) - 1, int(percentile * len(ordered)))]

    out = {}
    for rxn in reactions:
        stoich = {m.id: c for m, c in rxn.metabolites.items()}
        subs = [m for m, c in stoich.items() if c < 0]
        prods = [m for m, c in stoich.items() if c > 0]
        minor = {m for m in list(subs) + list(prods) if degree.get(m, 0) >= cutoff}
        keep_s = [m for m in subs if m not in minor] or subs
        keep_p = [m for m in prods if m not in minor] or prods
        out[rxn.id] = {(a, b) for a in keep_s for b in keep_p}
    return out


def evaluate_metdraw(model, drawn, percentile=0.90):
    """Precision and recall of MetDraw's drawn connections against KEGG's."""
    compound_of = {}
    for met in model.metabolites:
        ids = _annotation(met, "kegg.compound")
        if ids:
            compound_of[met.id] = set(ids)

    pairs = metdraw_pairs(model, list(model.reactions), percentile)
    considered = covered = 0
    drawn_edges = 0
    correct_edges = 0
    for rxn in model.reactions:
        target = set()
        for rid in _annotation(rxn, "kegg.reaction"):
            target |= drawn.get(rid, set())
        if not target:
            continue
        candidate = pairs.get(rxn.id) or set()
        usable = [(a, b) for a, b in candidate
                  if compound_of.get(a) and compound_of.get(b)]
        if not usable:
            continue
        considered += 1
        drawn_edges += len(usable)
        hit = False
        for a, b in usable:
            if any(frozenset((x, y)) in target
                   for x in compound_of[a] for y in compound_of[b]):
                correct_edges += 1
                hit = True
        if hit:
            covered += 1
    return {
        "reactions": considered,
        "recall": covered / considered if considered else float("nan"),
        "precision": correct_edges / drawn_edges if drawn_edges else float("nan"),
        "edges_per_reaction": drawn_edges / considered if considered else float("nan"),
    }


def evaluate(model, drawn, variant):
    """Agreement between MetaCarto's backbone and KEGG's drawn connection."""
    compound_of = {}
    for met in model.metabolites:
        ids = _annotation(met, "kegg.compound")
        if ids:
            compound_of[met.id] = set(ids)

    chosen = chosen_pairs(model, list(model.reactions), variant)

    considered = agree = 0
    missed = []
    for rxn in model.reactions:
        rids = _annotation(rxn, "kegg.reaction")
        target = set()
        for rid in rids:
            target |= drawn.get(rid, set())
        if not target:
            continue
        pair = chosen.get(rxn.id)
        if pair is None:
            continue
        a, b = pair
        ca, cb = compound_of.get(a), compound_of.get(b)
        if not ca or not cb:
            continue
        considered += 1
        if any(frozenset((x, y)) in target for x in ca for y in cb):
            agree += 1
        elif len(missed) < 12:
            missed.append((rxn.id, a, b))
    return considered, agree, missed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="iJO1366")
    parser.add_argument("--model-dir", default=os.path.join("data", "bigg", "models"))
    parser.add_argument("--kegg", default=os.path.join("data", "kegg"))
    parser.add_argument("--variants",
                        default="full,no_cofactor_tiering,no_cofactor_handling")
    parser.add_argument("--out", default=os.path.join("benchmarks", "results"))
    args = parser.parse_args(argv)

    import cobra
    path = os.path.join(args.model_dir, args.model + ".xml")
    if not os.path.exists(path):
        parser.error("no model at %s" % path)

    print("reading KGML reaction drawings ...")
    drawn = kegg_drawn_pairs(args.kegg)
    print("  %d KEGG reactions with a drawn substrate/product pair" % len(drawn))

    print("loading %s ..." % path)
    model = cobra.io.read_sbml_model(path)

    results = {}
    print()
    print("%-24s%12s%12s%10s" % ("variant", "reactions", "agree", "rate"))
    for variant in args.variants.split(","):
        considered, agree, missed = evaluate(model, drawn, variant)
        rate = agree / considered if considered else float("nan")
        results[variant] = {"considered": considered, "agree": agree,
                            "rate": rate, "examples_missed": missed}
        print("%-24s%12d%12d%9.1f%%" % (variant, considered, agree, 100 * rate))

    print()
    print("MetDraw's rule (connectivity cloning, no chemistry, no curated list)")
    print("%-24s%12s%12s%12s%14s" % ("threshold", "reactions", "recall",
                                     "precision", "edges/rxn"))
    md = {}
    for pct in (0.80, 0.90, 0.95):
        stats = evaluate_metdraw(model, drawn, pct)
        md["p%d" % int(pct * 100)] = stats
        print("%-24s%12d%11.1f%%%11.1f%%%14.2f"
              % ("top %d%% cloned" % int((1 - pct) * 100), stats["reactions"],
                 100 * stats["recall"], 100 * stats["precision"],
                 stats["edges_per_reaction"]))
    print("MetaCarto draws exactly one connection per reaction, so its recall")
    print("and precision are both the agreement rate above.")

    if "full" in results and results["full"]["examples_missed"]:
        print("\nreactions where the full method disagrees with KEGG's drawing:")
        for rid, a, b in results["full"]["examples_missed"][:8]:
            print("  %-16s drew %s -> %s" % (rid, strip_compartment(a),
                                             strip_compartment(b)))

    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        sha = "unknown"
    os.makedirs(args.out, exist_ok=True)
    target = os.path.join(args.out, "correctness_%s.json" % args.model)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump({"commit": sha, "model": args.model,
                   "kegg_reactions_drawn": len(drawn),
                   "variants": {k: {kk: vv for kk, vv in v.items()
                                    if kk != "examples_missed"}
                                for k, v in results.items()},
                   "metdraw": md},
                  handle, indent=2)
    print("\nwrote %s (commit %s)" % (target, sha))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
