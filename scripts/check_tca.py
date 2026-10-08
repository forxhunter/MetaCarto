"""Check that the TCA cycle is drawn as the TCA cycle, end to end.

    python scripts/check_tca.py e_coli_core iML1515 path/to/model.json
    python scripts/check_tca.py MitoMammal.json --reaction PDHm,CSm,...   # a DIY selection

What the cycle *should* be is worked out here from the model's stoichiometry,
not from the layout code being checked: in each compartment, eight anchor
compounds in textbook order, each consecutive pair joined by a reaction that
converts one into the other (directly, or through one intermediate such as
cis-aconitate), and every step able to run forwards. That is compared against
what the production path (decompose -> layout -> Escher JSON) emits:

  cluster    every reaction on the ring is in one cluster, and one pathway
             entry of the map header lists them
  ring       the drawn ring is those anchors, in that order, in one compartment
  closed     at every member the incoming and outgoing steps meet at one node
  round      members on a circle, in ring order around it
  arcs       each step drawn on its own stretch of the circle, not across it
  chords     nothing drawn across the ring but a real conversion between two
             members (the glyoxylate shunt), and no transporter drawn as a
             conversion between two different compounds
  middle     nothing else inside the ring: no line through it, no cofactor
  clean      no text overlaps, no unrelated reactions drawn on each other

Exit status is 1 if any model fails a check.
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Anchors: BiGG species, KEGG id, names. Kept separate from motifs.CANONICAL_CYCLES
# on purpose, so the check does not share a mistake with the code it checks.
TCA = (
    ("oaa", "C00036", ("oxaloacetate", "oxalacetate")),
    ("cit", "C00158", ("citrate", "citric acid")),
    ("icit", "C00311", ("isocitrate", "d-threo-isocitrate", "isocitric acid")),
    ("akg", "C00026", ("2-oxoglutarate", "alpha-ketoglutarate", "2-oxoglutaric acid",
                       "oxoglutarate")),
    ("succoa", "C00091", ("succinyl-coa", "succinyl coenzyme a")),
    ("succ", "C00042", ("succinate", "succinic acid")),
    ("fum", "C00122", ("fumarate", "fumaric acid")),
    ("mal__L", "C00149", ("l-malate", "malate", "(s)-malate")),
)
ANCHOR_NAMES = ("oxaloacetate", "citrate", "isocitrate", "2-oxoglutarate",
                "succinyl-CoA", "succinate", "fumarate", "malate")


# --------------------------------------------------------------------------
# the expected cycle, from stoichiometry
# --------------------------------------------------------------------------

def _anchor_of(met_id):
    from src.layout import identity
    for i, (species, kegg, names) in enumerate(TCA):
        if (identity.species(met_id) == species or kegg in identity.kegg(met_id)
                or identity.name(met_id) in names):
            return i
    return None


def _is_transport_of(reaction, met_id):
    """True if `reaction` moves this compound between compartments."""
    from src.layout import identity
    species = identity.species(met_id)
    return sum(1 for m in reaction.metabolites if identity.species(m.id) == species) > 1


def conversions(reactions):
    """{frozenset(x, y): {reaction ids}} for compounds on opposite sides of a
    reaction, in one compartment, of different species, neither transported."""
    from src.layout import identity
    out = {}
    for r in reactions:
        mets = [(m.id, c) for m, c in r.metabolites.items()]
        if len(mets) < 2 or all(c < 0 for _, c in mets) or all(c > 0 for _, c in mets):
            continue
        for x, cx in mets:
            for y, cy in mets:
                if x >= y or cx * cy >= 0:
                    continue
                if identity.compartment(x) != identity.compartment(y):
                    continue
                if identity.species(x) == identity.species(y):
                    continue
                if _is_transport_of(r, x) or _is_transport_of(r, y):
                    continue
                out.setdefault(frozenset((x, y)), set()).add(r.id)
    return out


def can_run(reaction, x, y):
    """True if `reaction` may carry flux converting x into y."""
    cx, cy = reaction.metabolites_by_id[x], reaction.metabolites_by_id[y]
    if reaction.lower_bound < 0 < reaction.upper_bound:
        return True
    forward = cx < 0 < cy
    return forward if reaction.upper_bound > 0 else not forward


def expected_cycles(reactions):
    """[{compartment, anchors: [ids], arcs: {(x, y): {rids}}, via: {...}}].

    One per compartment holding all eight anchors with every step present and
    runnable in the textbook direction. `arcs` maps each consecutive anchor
    pair to the reactions converting it directly, `via` to (intermediate,
    first-half rids, second-half rids) options.
    """
    from src.layout import identity

    by_id = {r.id: r for r in reactions}
    for r in reactions:
        r.metabolites_by_id = {m.id: c for m, c in r.metabolites.items()}
    conv = conversions(reactions)
    nbrs = {}
    for pair in conv:
        x, y = tuple(pair)
        nbrs.setdefault(x, set()).add(y)
        nbrs.setdefault(y, set()).add(x)

    anchors = {}
    for met in {m.id for r in reactions for m in r.metabolites}:
        i = _anchor_of(met)
        if i is not None and identity.currency(met) is None:
            anchors.setdefault(identity.compartment(met), {}).setdefault(i, []).append(met)

    found = []
    for comp, present in sorted(anchors.items()):
        if len(present) < len(TCA):
            continue
        ids = [sorted(present[i])[0] for i in range(len(TCA))]
        anchor_set = set(ids)
        arcs, via, missing = {}, {}, []
        for i, x in enumerate(ids):
            y = ids[(i + 1) % len(ids)]
            direct = {rid for rid in conv.get(frozenset((x, y)), ())
                      if can_run(by_id[rid], x, y)}
            # An intermediate only where there is no direct step: cis-aconitate
            # when aconitase is written as two half-reactions. With a direct
            # step present, malate -> pyruvate -> oxaloacetate (malic enzyme,
            # then pyruvate carboxylase) is a bypass, not part of the cycle.
            options = []
            for z in ([] if direct else sorted(nbrs.get(x, set()) & nbrs.get(y, set()))):
                if z in anchor_set or identity.currency(z) is not None:
                    continue
                first = {rid for rid in conv[frozenset((x, z))] if can_run(by_id[rid], x, z)}
                second = {rid for rid in conv[frozenset((z, y))] if can_run(by_id[rid], z, y)}
                if first and second:
                    options.append((z, first, second))
            if direct:
                arcs[(x, y)] = direct
            if options:
                via[(x, y)] = options
            if not direct and not options:
                missing.append(f"{ANCHOR_NAMES[i]} -> {ANCHOR_NAMES[(i + 1) % len(ids)]}")
        found.append({"compartment": comp, "anchors": ids, "arcs": arcs, "via": via,
                      "missing": missing})
    return found


def arc_reactions(cycle):
    out = set()
    for rids in cycle["arcs"].values():
        out |= rids
    for options in cycle["via"].values():
        for _, first, second in options:
            out |= first | second
    return out


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------

class Report:
    def __init__(self, title):
        self.title = title
        self.rows = []

    def check(self, name, ok, detail=""):
        self.rows.append((name, bool(ok), detail))
        return ok

    def info(self, name, detail):
        self.rows.append((name, None, detail))

    @property
    def failed(self):
        return any(ok is False for _, ok, _ in self.rows)

    def print(self):
        print(f"\n== {self.title}")
        for name, ok, detail in self.rows:
            mark = {True: "ok  ", False: "FAIL", None: "    "}[ok]
            print(f"  {mark} {name:10s} {detail}")


def _angle(p, c):
    return math.atan2(p[1] - c[1], p[0] - c[0])


def _angdiff(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def _circular_mean(a, b):
    return math.atan2(math.sin(a) + math.sin(b), math.cos(a) + math.cos(b))


def _reaction_nodes(body, key):
    """(metabolite node ids by bigg id, midmarker ids, all segment polylines)."""
    nodes = body["nodes"]
    mets, mids = {}, []
    for seg in body["reactions"][key]["segments"].values():
        for nid in (seg["from_node_id"], seg["to_node_id"]):
            node = nodes.get(nid)
            if node is None:
                continue
            if node["node_type"] == "metabolite":
                mets.setdefault(node["bigg_id"], set()).add(nid)
            elif node["node_type"] == "midmarker":
                mids.append(nid)
    return mets, mids


def _ink(body, key, steps=12):
    from src.layout.metrics import _bezier_points
    nodes = body["nodes"]
    for seg in body["reactions"][key]["segments"].values():
        a, b = nodes.get(seg["from_node_id"]), nodes.get(seg["to_node_id"])
        if a is None or b is None:
            continue
        p0, p3 = (a["x"], a["y"]), (b["x"], b["y"])
        if seg.get("b1") and seg.get("b2"):
            pts = list(_bezier_points(p0, (seg["b1"]["x"], seg["b1"]["y"]),
                                      (seg["b2"]["x"], seg["b2"]["y"]), p3, steps))
        else:
            pts = [(p0[0] + (p3[0] - p0[0]) * t / steps, p0[1] + (p3[1] - p0[1]) * t / steps)
                   for t in range(steps + 1)]
        yield seg, pts


def check_map(report, model, reactions, escher_map, result, cycle):
    """The drawing checks, for one expected cycle on one emitted map."""
    from src.layout import identity
    from src.layout.metrics import reaction_overlaps, score

    body = escher_map[1]
    cg = result.cgraph
    by_id = {r.id: r for r in reactions}
    key_of = {}
    for key, rx in body["reactions"].items():
        key_of.setdefault(rx["bigg_id"], []).append(key)
    anchors = cycle["anchors"]

    # ---- ring: the anchors, in order, as one drawn ring ----------------------
    ring = next((list(r) for r in result.rings if anchors[0] in r), None)
    if not report.check("ring", ring is not None,
                        "no drawn ring contains " + anchors[0] if ring is None else ""):
        return
    order = [m for m in ring if m in anchors]
    k = len(anchors)
    start = order.index(anchors[0]) if anchors[0] in order else 0
    rotated = order[start:] + order[:start]
    textbook = rotated == anchors
    backwards = ([rotated[0]] + rotated[1:][::-1]) == anchors
    extras = [m for m in ring if m not in anchors]
    allowed = {z for opts in cycle["via"].values() for z, _, _ in opts}
    report.check("ring", (textbook or backwards) and len(order) == k
                 and set(extras) <= allowed
                 and len({identity.compartment(m) for m in ring}) == 1,
                 f"{len(ring)} members: " + " -> ".join(ring))

    # Steps of the drawn ring: consecutive drawn members, and the reactions
    # converting each pair (from stoichiometry, restricted to this map).
    conv = conversions(reactions)
    steps = []
    for i, x in enumerate(ring):
        y = ring[(i + 1) % len(ring)]
        rids = {rid for rid in conv.get(frozenset((x, y)), ()) if rid in key_of}
        steps.append((x, y, rids))
    empty = [f"{x}->{y}" for x, y, rids in steps if not rids]
    report.check("ring", not empty, "steps with no reaction: " + ", ".join(empty) if empty else "")

    # A step's reaction is "on" the step when the layout made the pair its
    # main pair. At least one runnable reaction per step must be.
    on_step = {}
    for x, y, rids in steps:
        on = sorted(rid for rid in rids
                    if rid in cg.reactions
                    and {cg.reactions[rid].main_sub, cg.reactions[rid].main_prod} == {x, y})
        on_step[(x, y)] = on
        runnable = [rid for rid in on if can_run(by_id[rid], x, y) or can_run(by_id[rid], y, x)]
        report.check("ring", bool(runnable),
                     f"{x}->{y}: {', '.join(on) or 'NOTHING on this step'}"
                     + ("" if any(can_run(by_id[r], x, y) for r in on) else
                        "  (no reaction on it can run forwards)"))
        off = sorted(rids - set(on))
        if off:
            report.info("ring", f"{x}->{y}: also converted by {', '.join(off)}, "
                        "drawn on another main pair")

    # ---- closed: one node per member, shared by both of its steps ------------
    member_nodes = {m: set() for m in ring}
    step_keys = {}
    for x, y, _ in steps:
        keys = [key for rid in on_step[(x, y)] for key in key_of[rid]]
        step_keys[(x, y)] = keys
        for key in keys:
            mets, _ = _reaction_nodes(body, key)
            for m in (x, y):
                member_nodes[m] |= {n for n in mets.get(m, ())
                                    if body["nodes"][n].get("node_is_primary", True)}
    split = {m: sorted(n) for m, n in member_nodes.items() if len(n) != 1}
    if not report.check("closed", not split,
                        "members drawn as more than one node (or none): %s" % split if split
                        else f"{len(ring)} members, one node each"):
        return
    node_of = {m: next(iter(n)) for m, n in member_nodes.items()}
    pts = {m: (body["nodes"][n]["x"], body["nodes"][n]["y"]) for m, n in node_of.items()}

    # ---- round ---------------------------------------------------------------
    cx = sum(p[0] for p in pts.values()) / len(pts)
    cy = sum(p[1] for p in pts.values()) / len(pts)
    radii = [math.hypot(p[0] - cx, p[1] - cy) for p in pts.values()]
    R = sum(radii) / len(radii)
    cv = (sum((r - R) ** 2 for r in radii) / len(radii)) ** 0.5 / R
    report.check("round", cv < 0.10, f"radius {R:.0f}, radial CV {cv:.3f}")
    angles = [_angle(pts[m], (cx, cy)) for m in ring]
    turns = [_angdiff(b, a) for a, b in zip(angles, angles[1:] + angles[:1])]
    one_way = all(t > 0 for t in turns) or all(t < 0 for t in turns)
    report.check("round", one_way and abs(abs(sum(turns)) - 2 * math.pi) < 1e-6,
                 "members in ring order around the circle" if one_way
                 else f"order around the circle broken: turns {[round(math.degrees(t)) for t in turns]}")

    # ---- arcs: each step's midmarker on its own stretch of the circle --------
    bad = []
    n = len(ring)
    for (x, y), keys in step_keys.items():
        mid_angle = _circular_mean(_angle(pts[x], (cx, cy)), _angle(pts[y], (cx, cy)))
        for key in keys:
            _, mids = _reaction_nodes(body, key)
            for nid in mids:
                node = body["nodes"][nid]
                r = math.hypot(node["x"] - cx, node["y"] - cy)
                off = abs(_angdiff(_angle((node["x"], node["y"]), (cx, cy)), mid_angle))
                # Parallel reactions on one step bow outward in lanes.
                if not (0.6 * R <= r <= 1.0 * R + 120.0 * len(keys) and off <= math.pi / n):
                    bad.append(f"{body['reactions'][key]['bigg_id']} (r={r / R:.2f}R, "
                               f"{math.degrees(off):.0f} deg off its arc)")
    report.check("arcs", not bad, "; ".join(bad) if bad else
                 f"{sum(len(k) for k in step_keys.values())} step reactions on their arcs")

    # ---- chords --------------------------------------------------------------
    members = set(ring)
    adjacent = {frozenset((x, y)) for x, y, _ in steps}
    on_ring = {key for keys in step_keys.values() for key in keys}
    legit_chords, wrong = set(), []
    for rid, rec in cg.reactions.items():
        pair = {rec.main_sub, rec.main_prod}
        if rid not in key_of or None in pair:
            continue
        r = by_id.get(rid)
        if r is None:
            continue
        r.metabolites_by_id = {m.id: c for m, c in r.metabolites.items()}
        x, y = rec.main_sub, rec.main_prod
        transported = [m for m in (x, y) if _is_transport_of(r, m)]
        if identity.species(x) != identity.species(y) and transported:
            wrong.append(f"{rid} drawn {x} -> {y}, but it carries "
                         f"{', '.join(identity.species(m) for m in transported)} across a membrane")
        elif pair <= members and frozenset(pair) not in adjacent:
            if frozenset(pair) in conv:
                legit_chords.update(key_of[rid])
            else:
                wrong.append(f"{rid} drawn across the ring {x} -> {y}, "
                             "which it does not convert")
    report.check("chords", not wrong, "; ".join(wrong) if wrong else
                 (f"real chords: {sorted({body['reactions'][k]['bigg_id'] for k in legit_chords})}"
                  if legit_chords else "none"))

    # Anything else drawn through the ring's interior.
    inside, stubs_inside = [], 0
    for key, rx in body["reactions"].items():
        if key in legit_chords:
            continue
        for seg, line in _ink(body, key):
            ends = {seg["from_node_id"], seg["to_node_id"]}
            hit = any(math.hypot(px - cx, py - cy) < 0.7 * R for px, py in line)
            if not hit:
                continue
            if key in on_ring:
                stubs_inside += 1
            else:
                inside.append(rx["bigg_id"])
                break
    # A curated TCA cycle is a circle with nothing inside (render.py says the
    # same): the only line across it is a real chord such as the glyoxylate
    # shunt.
    report.check("middle", not inside,
                 f"drawn through the ring's interior: {sorted(set(inside))}" if inside else
                 "no other reaction drawn through the interior")
    report.check("middle", not stubs_inside,
                 f"{stubs_inside} cofactor segments of ring steps reach inside the ring"
                 if stubs_inside else "every ring step's cofactors face outward")

    # ---- header --------------------------------------------------------------
    pathways = escher_map[0].get("pathways") or []
    ring_rids = {body["reactions"][k]["bigg_id"] for k in on_ring}
    holders = [p["name"] for p in pathways if ring_rids & set(
        body["reactions"][k]["bigg_id"] for k in p["reactions"] if k in body["reactions"])
        or ring_rids & set(p["reactions"])]
    report.check("cluster", len(holders) == 1,
                 f"map header files the ring steps under {holders}")

    # ---- clean ---------------------------------------------------------------
    values = score(escher_map)
    overlaps = reaction_overlaps(escher_map)
    text = {k: values[k] for k in ("label_overlaps", "label_on_node", "label_on_edge")}
    unrelated = {k: v for k, v in overlaps.items() if k.endswith("_unrelated")}
    report.check("clean", not any(text.values()) and not any(unrelated.values()),
                 f"text {text}, unrelated {unrelated}")


def _save(escher_map, out, name):
    if not out:
        return
    from src.layout import preview, render
    os.makedirs(out, exist_ok=True)
    stem = os.path.join(out, "".join(c if c.isalnum() or c in "-_" else "_" for c in name))
    render.save(escher_map, stem + ".json")
    preview.render(escher_map, stem + ".png")


def check_model(path, selection=None, use_fba=True, out=None):
    import layout_v2
    from src.layout import decompose, identity, render
    from src.layout.compound import compute_cofactor_scores
    from src.layout.engine import layout_reactions

    model = layout_v2.load_model(path)
    title = layout_v2.model_stem(path) + (" (selection)" if selection else "")
    report = Report(title)
    reactions = list(model.reactions)

    expected = expected_cycles(reactions)
    complete = [c for c in expected if not c["missing"]]
    for c in expected:
        if c["missing"]:
            report.info("expected", f"[{c['compartment']}] no complete cycle; "
                        f"missing {', '.join(c['missing'])}")
    if not complete and not selection:
        report.info("expected", "no complete TCA cycle in this model; nothing to draw round")
        return report

    scores = compute_cofactor_scores(model)
    if selection:
        chosen = [model.reactions.get_by_id(rid) for rid in selection]
        expected_sel = [c for c in expected_cycles(chosen) if not c["missing"]]
        for c in expected_cycles(chosen):
            if c["missing"]:
                report.info("expected", f"[{c['compartment']}] the selection has no complete "
                            f"cycle; missing {', '.join(c['missing'])}")
        catalogue_groups = decompose.clusters(model, scores)
        pathway_of = {r.id: n for n, rs in catalogue_groups.items() for r in rs}
        parts = {}
        for r in chosen:
            parts.setdefault(pathway_of.get(r.id, "Selected reactions"), []).append(r)
        result = layout_reactions(model, chosen, title, use_fba=use_fba)
        render.annotate_pathways(result.escher_map, sorted(parts.items()))
        _save(result.escher_map, out, title)
        if not expected_sel:
            # Nothing should be drawn round; check the header at least.
            holders = [p["name"] for p in result.escher_map[0].get("pathways", [])]
            report.check("cluster", len(holders) == 1,
                         f"map header files the selection under {holders}")
            report.check("ring", not any(set(r) & {a for c in expected for a in c["anchors"]}
                                         for r in result.rings),
                         "no ring drawn, as expected without a complete cycle")
            return report
        for cycle in expected_sel:
            report.info("expected", f"[{cycle['compartment']}] " + " -> ".join(cycle["anchors"]))
            check_map(report, model, chosen, result.escher_map, result, cycle)
        return report

    groups = decompose.clusters(model, scores)
    owner = {r.id: name for name, rs in groups.items() for r in rs}
    for cycle in complete:
        report.info("expected", f"[{cycle['compartment']}] " + " -> ".join(cycle["anchors"]))
        rids = arc_reactions(cycle)
        homes = {}
        for rid in sorted(rids):
            homes.setdefault(owner.get(rid, "(no cluster)"), []).append(rid)
        home = max(homes, key=lambda n: len(homes[n]))
        # Every step needs a reaction that runs it forwards in the home
        # cluster. Other converters of a step may live elsewhere: SCOT turns
        # succinyl-CoA into succinate too, but it is ketone-body metabolism.
        inside = {r.id for r in groups[home]}
        uncovered = []
        for i, x in enumerate(cycle["anchors"]):
            y = cycle["anchors"][(i + 1) % len(cycle["anchors"])]
            direct = cycle["arcs"].get((x, y), set())
            split = cycle["via"].get((x, y), [])
            if not (direct & inside or any(a & inside and b & inside for _, a, b in split)):
                uncovered.append(f"{x}->{y} ({', '.join(sorted(direct)) or 'split'} "
                                 f"in {sorted({owner.get(r) for r in direct})})")
        report.check("cluster", not uncovered,
                     f"every step has its reaction in '{home}'" if not uncovered else
                     f"steps missing from '{home}': {'; '.join(uncovered)}")
        # Cluster membership is decided on the whole-model graph, where a
        # reaction can be paired differently from its drawing in the cluster.
        # MitoMammal's succinate/malate carrier was paired malate -> succinate
        # there, so ring closure pulled it into the TCA map, where it was then
        # drawn correctly as succinate crossing the membrane.
        whole = decompose._whole_graph(model)
        fake = []
        for r in groups[home]:
            rec = whole.reactions.get(r.id)
            if rec is None or rec.main_sub is None:
                continue
            pair = (rec.main_sub, rec.main_prod)
            moved = [m for m in pair if _is_transport_of(r, m)]
            if identity.species(pair[0]) != identity.species(pair[1]) and moved:
                fake.append(f"{r.id} ({pair[0]} -> {pair[1]})")
        report.check("cluster", not fake,
                     "carriers paired as conversions on the whole model: " + ", ".join(fake)
                     if fake else "no carrier paired as a conversion on the whole model")
        strays = {n: v for n, v in homes.items() if n != home}
        if strays:
            report.info("cluster", f"other converters of a step, kept elsewhere: {strays}")
        result = layout_reactions(model, groups[home], home, use_fba=use_fba)
        render.annotate_pathways(result.escher_map, [(home, groups[home])])
        _save(result.escher_map, out, f"{title}_{home}")
        check_map(report, model, groups[home], result.escher_map, result, cycle)
    return report


def resolve(spec):
    if os.path.exists(spec):
        return spec
    for ext in (".json", ".xml"):
        path = os.path.join("data", "bigg", "models", spec + ext)
        if os.path.exists(path):
            return path
    raise SystemExit(f"model not found: {spec}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="+")
    parser.add_argument("--reaction", help="check a DIY selection instead: ids, comma separated")
    parser.add_argument("--no-fba", action="store_true")
    parser.add_argument("--out", help="also write each checked map here (JSON and PNG)")
    args = parser.parse_args(argv)
    selection = args.reaction.split(",") if args.reaction else None

    failed = []
    for spec in args.models:
        try:
            report = check_model(resolve(spec), selection, use_fba=not args.no_fba,
                                 out=args.out)
        except Exception as exc:                       # keep going, but say so
            import traceback
            traceback.print_exc()
            print(f"\n== {spec}\n  FAIL crashed: {exc}")
            failed.append(spec)
            continue
        report.print()
        if report.failed:
            failed.append(report.title)
    print(f"\n{len(args.models) - len(failed)} of {len(args.models)} passed"
          + (f"; failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
