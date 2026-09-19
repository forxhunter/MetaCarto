"""How hard is the KEGG agreement test, and what does chance score on it?

The agreement measure asks whether the pair \\mytool draws is one of the
substrate--product connections the curated KGML drawing contains for that
reaction. A KGML <reaction> element carries several <substrate> and several
<product> children, so the reference is a *set*:

    D(R) = { {s, p} : s in substrates(R), p in products(R) }

unioned over every curated drawing that contains R. Agreement is membership in
D(R), not equality with a single curated pair, and the paper has to say so.

Membership in a set is only a meaningful test if the set is small relative to
the choices available. This measures both sides of that:

  |D(R)|      how many pairs the reference admits
  |C(R)|      how many substrate--product pairs of the BiGG reaction are
              candidates at all, i.e. have KEGG compound identifiers
  chance      |C(R) inter D(R)| / |C(R)|, the probability that picking
              uniformly at random from the candidates lands in the reference

The mean of `chance` over the scored reactions is the number \\mytool's
agreement rate has to be read against. Without it, a high agreement rate could
mean the method is right or merely that the test is easy.

    python -m src.bench.correctness_control --models iJO1366,iMM904,iYO844,iAF692
"""

import argparse
import json
import os
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from src.bench.correctness import (_annotation, chosen_pairs,  # noqa: E402
                                   kegg_drawn_pairs)


def analyse(model, drawn):
    """Per-reaction difficulty of the agreement test, and what chance scores."""
    compound_of = {}
    for met in model.metabolites:
        ids = _annotation(met, "kegg.compound")
        if ids:
            compound_of[met.id] = set(ids)

    chosen = chosen_pairs(model, list(model.reactions), "full")

    rows = []
    for rxn in model.reactions:
        target = set()
        for rid in _annotation(rxn, "kegg.reaction"):
            target |= drawn.get(rid, set())
        if not target:
            continue
        pair = chosen.get(rxn.id)
        if pair is None:
            continue
        a, b = pair
        if not compound_of.get(a) or not compound_of.get(b):
            continue

        # The candidate space the choice was actually made from: every
        # substrate/product combination of this reaction whose two members
        # both carry a KEGG compound id.
        subs = [m.id for m, c in rxn.metabolites.items()
                if c < 0 and m.id in compound_of]
        prods = [m.id for m, c in rxn.metabolites.items()
                 if c > 0 and m.id in compound_of]
        candidates = set()
        hits = set()
        for s in subs:
            for p in prods:
                if s == p:
                    continue
                candidates.add(frozenset((s, p)))
                if any(frozenset((x, y)) in target
                       for x in compound_of[s] for y in compound_of[p]):
                    hits.add(frozenset((s, p)))
        if not candidates:
            continue

        agreed = any(frozenset((x, y)) in target
                     for x in compound_of[a] for y in compound_of[b])
        rows.append({
            "reaction": rxn.id,
            "reference_pairs": len(target),
            "candidates": len(candidates),
            "candidates_in_reference": len(hits),
            "chance": len(hits) / len(candidates),
            "agreed": agreed,
        })
    return rows


def summarise(rows):
    if not rows:
        return {}
    chance = [r["chance"] for r in rows]
    cands = [r["candidates"] for r in rows]
    ref = [r["reference_pairs"] for r in rows]
    agreed = sum(1 for r in rows if r["agreed"])
    # The reactions where the test can actually discriminate: more than one
    # candidate, and not every candidate in the reference.
    hard = [r for r in rows if r["candidates"] > 1 and r["chance"] < 1.0]
    return {
        "reactions": len(rows),
        "agreement": agreed / len(rows),
        "chance_mean": statistics.mean(chance),
        "chance_median": statistics.median(chance),
        "candidates_median": statistics.median(cands),
        "candidates_max": max(cands),
        "reference_median": statistics.median(ref),
        "reference_max": max(ref),
        "trivial_reactions": sum(1 for r in rows if r["chance"] >= 1.0),
        "single_candidate": sum(1 for r in rows if r["candidates"] == 1),
        "discriminating": {
            "reactions": len(hard),
            "agreement": (sum(1 for r in hard if r["agreed"]) / len(hard)
                          if hard else None),
            "chance_mean": statistics.mean([r["chance"] for r in hard])
            if hard else None,
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="iJO1366,iMM904,iYO844,iAF692")
    parser.add_argument("--model-dir", default=os.path.join("data", "bigg", "models"))
    parser.add_argument("--kegg", default=os.path.join("data", "kegg"))
    parser.add_argument("--out", default=os.path.join("benchmarks", "results"))
    args = parser.parse_args(argv)

    import cobra

    print("reading KGML reaction drawings ...")
    drawn = kegg_drawn_pairs(args.kegg)
    print("  %d KEGG reactions with at least one drawn pair" % len(drawn))
    sizes = [len(v) for v in drawn.values()]
    print("  pairs per reaction: median %d, 90th pct %d, max %d"
          % (statistics.median(sizes), sorted(sizes)[int(0.9 * len(sizes))],
             max(sizes)))

    out = {}
    print()
    print("%-12s%8s%11s%10s%12s%12s" % ("model", "rxns", "agreement", "chance",
                                        "cand med", "trivial"))
    for name in args.models.split(","):
        name = name.strip()
        path = None
        for ext in (".xml", ".json"):
            candidate = os.path.join(args.model_dir, name + ext)
            if os.path.exists(candidate):
                path = candidate
                break
        if path is None:
            print("%-12s (no model file)" % name)
            continue
        model = (cobra.io.load_json_model(path) if path.endswith(".json")
                 else cobra.io.read_sbml_model(path))
        rows = analyse(model, drawn)
        s = summarise(rows)
        out[name] = s
        if not s:
            continue
        print("%-12s%8d%10.1f%%%9.1f%%%12d%11.1f%%"
              % (name, s["reactions"], 100 * s["agreement"],
                 100 * s["chance_mean"], s["candidates_median"],
                 100 * s["trivial_reactions"] / s["reactions"]))

    print()
    print("on the reactions where the test can discriminate")
    print("  (more than one candidate, and not every candidate in the reference)")
    print("%-12s%8s%11s%10s" % ("model", "rxns", "agreement", "chance"))
    for name, s in out.items():
        d = s.get("discriminating") or {}
        if d.get("reactions"):
            print("%-12s%8d%10.1f%%%9.1f%%"
                  % (name, d["reactions"], 100 * d["agreement"],
                     100 * d["chance_mean"]))

    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      stderr=subprocess.DEVNULL).decode().strip()
    except Exception:                                   # noqa: BLE001
        sha = "unknown"
    os.makedirs(args.out, exist_ok=True)
    target = os.path.join(args.out, "correctness_control.json")
    with open(target, "w", encoding="utf-8") as handle:
        json.dump({"commit": sha, "models": out}, handle, indent=2)
    print("\nwrote %s (commit %s)" % (target, sha))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
