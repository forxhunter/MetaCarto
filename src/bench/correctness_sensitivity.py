"""Does the KEGG agreement rate survive a stricter reference definition?

Agreement is membership in D(R), the union of substrate-product pairs over
every KGML <reaction> element carrying R, over every file. A union is the
permissive choice, and the headline rate is only worth reporting if it does
not depend on that choice. This rescores the same reactions under references
built more strictly:

    union         as shipped: every pair from every file
    intersection  only pairs drawn in *every* file that contains R
    majority      pairs drawn in more than half of those files
    single_r      only elements whose name attribute holds one R number,
                  so a pair set is never attributed to a reaction that
                  merely shares an element
    one_to_one    only elements listing exactly one substrate and one
                  product, which is 90.5% of them and the case where the
                  reference is unambiguous by construction

KEGG reuses one reference drawing across organisms, so the files behind an R
number are near-duplicates rather than independent curators. Intersection is
therefore the strict bound, not a consensus measure, and is reported as such.

    python -m src.bench.correctness_sensitivity
"""

import argparse
import collections
import glob
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from src.bench.correctness import _annotation, chosen_pairs  # noqa: E402

VARIANTS = ("union", "majority", "intersection", "single_r", "one_to_one")


def per_file_pairs(directory):
    """{R: [set of pairs from one file, ...]} keeping files separate."""
    out = collections.defaultdict(list)
    singles = collections.defaultdict(set)
    one_one = collections.defaultdict(set)
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".xml"):
            continue
        try:
            root = ET.parse(os.path.join(directory, name)).getroot()
        except ET.ParseError:
            continue
        compound = {e.get("id"): (e.get("name") or "").replace("cpd:", "")
                    for e in root.findall("entry")
                    if e.get("type") == "compound"}
        here = collections.defaultdict(set)
        for reaction in root.findall("reaction"):
            names = (reaction.get("name") or "").replace("rn:", "").split()
            subs = [compound.get(s.get("id")) for s in reaction.findall("substrate")]
            prods = [compound.get(p.get("id")) for p in reaction.findall("product")]
            n_sub, n_prod = len(subs), len(prods)
            subs = [s for s in subs if s]
            prods = [p for p in prods if p]
            for rid in names:
                for s in subs:
                    for p in prods:
                        if s == p:
                            continue
                        pair = frozenset((s, p))
                        here[rid].add(pair)
                        if len(names) == 1:
                            singles[rid].add(pair)
                        if n_sub == 1 and n_prod == 1:
                            one_one[rid].add(pair)
        for rid, pairs in here.items():
            out[rid].append(pairs)
    return out, singles, one_one


def build_reference(per_file, singles, one_one, variant):
    if variant == "single_r":
        return dict(singles)
    if variant == "one_to_one":
        return dict(one_one)
    reference = {}
    for rid, sets in per_file.items():
        if variant == "union":
            reference[rid] = set().union(*sets)
        elif variant == "intersection":
            reference[rid] = set.intersection(*[set(s) for s in sets])
        else:                                    # majority
            tally = collections.Counter()
            for s in sets:
                tally.update(s)
            need = len(sets) / 2.0
            reference[rid] = {p for p, n in tally.items() if n > need}
    return reference


def score(model, reference, chosen, compound_of):
    considered = agree = 0
    for rxn in model.reactions:
        target = set()
        for rid in _annotation(rxn, "kegg.reaction"):
            target |= reference.get(rid, set())
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
    return considered, agree


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="iJO1366,iMM904,iYO844,iAF692")
    parser.add_argument("--model-dir", default=os.path.join("data", "bigg", "models"))
    parser.add_argument("--kegg", default=os.path.join("data", "kegg"))
    parser.add_argument("--out", default=os.path.join("benchmarks", "results"))
    args = parser.parse_args(argv)

    import cobra

    print("reading KGML, keeping files separate ...")
    per_file, singles, one_one = per_file_pairs(args.kegg)
    print("  %d R numbers; median %d file(s) each"
          % (len(per_file),
             sorted(len(v) for v in per_file.values())[len(per_file) // 2]))

    references = {v: build_reference(per_file, singles, one_one, v)
                  for v in VARIANTS}

    out = {}
    print()
    print("%-12s" % "model" + "".join("%14s" % v for v in VARIANTS))
    for name in args.models.split(","):
        name = name.strip()
        path = None
        for ext in (".xml", ".json"):
            candidate = os.path.join(args.model_dir, name + ext)
            if os.path.exists(candidate):
                path = candidate
                break
        if path is None:
            continue
        model = (cobra.io.load_json_model(path) if path.endswith(".json")
                 else cobra.io.read_sbml_model(path))
        compound_of = {}
        for met in model.metabolites:
            ids = _annotation(met, "kegg.compound")
            if ids:
                compound_of[met.id] = set(ids)
        chosen = chosen_pairs(model, list(model.reactions), "full")

        row = {}
        cells = []
        for variant in VARIANTS:
            considered, agree = score(model, references[variant], chosen,
                                      compound_of)
            rate = agree / considered if considered else None
            row[variant] = {"considered": considered, "agree": agree,
                            "rate": rate}
            cells.append("%13s" % ("%.1f%% (%d)" % (100 * rate, considered)
                                   if rate is not None else "--"))
        out[name] = row
        print("%-12s" % name + " ".join(cells))

    # Per model, not across them: the cost of a stricter reference is
    # union minus the worst strict variant *for the same model*. Taking the
    # best union rate of one model against the worst strict rate of another
    # gives a number twice as large and means nothing.
    strict_variants = ("intersection", "majority", "one_to_one")
    drops, strict_all, shipped = [], [], []
    for row in out.values():
        if row["union"]["rate"] is None:
            continue
        shipped.append(row["union"]["rate"])
        worst = min(row[v]["rate"] for v in strict_variants
                    if row[v]["rate"] is not None)
        strict_all.append(worst)
        drops.append(row["union"]["rate"] - worst)
    if drops:
        print()
        print("shipped (union):     %.1f-%.1f%%"
              % (100 * min(shipped), 100 * max(shipped)))
        print("worst strict variant: %.1f-%.1f%%"
              % (100 * min(strict_all), 100 * max(strict_all)))
        print("largest per-model drop: %.1f percentage points"
              % (100 * max(drops)))

    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      stderr=subprocess.DEVNULL).decode().strip()
    except Exception:                                   # noqa: BLE001
        sha = "unknown"
    os.makedirs(args.out, exist_ok=True)
    target = os.path.join(args.out, "correctness_sensitivity.json")
    with open(target, "w", encoding="utf-8") as handle:
        json.dump({"commit": sha, "variants": list(VARIANTS), "models": out},
                  handle, indent=2)
    print("\nwrote %s (commit %s)" % (target, sha))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
